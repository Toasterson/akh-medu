//! Memory-mapped file store for warm data.
//!
//! Uses `memmap2` for memory-mapped I/O, providing near-zero-cost reads
//! for data that is too large for RAM but frequently accessed.
//! Works on both Linux and illumos (POSIX mmap).

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use memmap2::MmapMut;

use crate::error::StoreError;
use crate::store::StoreResult;
use crate::symbol::SymbolId;

/// Header written at the start of the mmap file.
const HEADER_MAGIC: &[u8; 8] = b"AKHMMAP\0";
const HEADER_VERSION: u32 = 1;
const HEADER_SIZE: usize = 64; // Reserve 64 bytes for future expansion

/// Memory-mapped file store for persistent symbol data.
///
/// Data is written sequentially (append-only). An in-memory index maps
/// `SymbolId` to `(offset, length)` within the mmap region.
pub struct MmapStore {
    path: PathBuf,
    mmap: RwLock<MmapMut>,
    /// Index: SymbolId → (offset, length) in the mmap region.
    index: RwLock<HashMap<SymbolId, (usize, usize)>>,
    /// Current write position (next free byte after header).
    write_pos: RwLock<usize>,
    /// Total file/mmap capacity.
    capacity: usize,
}

impl MmapStore {
    /// Open an existing mmap store or create a new one.
    ///
    /// `name` is used as the filename stem (e.g., "symbols" → "symbols.mmap").
    pub fn open_or_create(data_dir: &Path, name: &str) -> StoreResult<Self> {
        fs::create_dir_all(data_dir).map_err(|e| StoreError::Io { source: e })?;

        let path = data_dir.join(format!("{name}.mmap"));
        let initial_size: u64 = 64 * 1024 * 1024; // 64 MB initial

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| StoreError::Io { source: e })?;

        let file_len = file
            .metadata()
            .map_err(|e| StoreError::Io { source: e })?
            .len();

        if file_len == 0 {
            // New file — set size and write header
            file.set_len(initial_size)
                .map_err(|e| StoreError::Io { source: e })?;
        }

        let actual_len = file
            .metadata()
            .map_err(|e| StoreError::Io { source: e })?
            .len() as usize;

        // Safety: we've created the file and own it exclusively at this point.
        let mut mmap = unsafe {
            MmapMut::map_mut(&file).map_err(|e| StoreError::Mmap {
                message: e.to_string(),
            })?
        };

        let (index, write_pos) = if file_len == 0 {
            // Initialize header
            mmap[..8].copy_from_slice(HEADER_MAGIC);
            mmap[8..12].copy_from_slice(&HEADER_VERSION.to_le_bytes());
            mmap.flush().map_err(|e| StoreError::Io { source: e })?;
            (HashMap::new(), HEADER_SIZE)
        } else {
            // Validate header and rebuild index
            if &mmap[..8] != HEADER_MAGIC {
                return Err(StoreError::Mmap {
                    message: "invalid mmap file header — is this an akh-medu data file?".into(),
                });
            }
            let version = u32::from_le_bytes(mmap[8..12].try_into().expect("4 bytes for version"));
            if version != HEADER_VERSION {
                return Err(StoreError::Mmap {
                    message: format!("mmap file version {version} != expected {HEADER_VERSION}"),
                });
            }
            // Rebuild index by scanning entries: each entry is [8-byte SymbolId][4-byte len][data]
            let mut index = HashMap::new();
            let mut pos = HEADER_SIZE;
            while pos + 12 <= actual_len {
                let id_bytes: [u8; 8] = mmap[pos..pos + 8].try_into().expect("8 bytes");
                let raw_id = u64::from_le_bytes(id_bytes);
                if raw_id == 0 {
                    break; // End of entries (uninitialized region)
                }
                let len_bytes: [u8; 4] = mmap[pos + 8..pos + 12].try_into().expect("4 bytes");
                let data_len = u32::from_le_bytes(len_bytes) as usize;
                if pos + 12 + data_len > actual_len {
                    break; // Truncated entry
                }
                if let Some(sym_id) = SymbolId::new(raw_id) {
                    index.insert(sym_id, (pos + 12, data_len));
                }
                pos += 12 + data_len;
            }
            (index, pos)
        };

        Ok(Self {
            path,
            mmap: RwLock::new(mmap),
            index: RwLock::new(index),
            write_pos: RwLock::new(write_pos),
            capacity: actual_len,
        })
    }

    /// Append a value to the mmap store.
    pub fn put(&self, key: SymbolId, value: &[u8]) -> StoreResult<()> {
        let entry_size = 12 + value.len(); // 8 (id) + 4 (len) + data
        let mut write_pos = self.write_pos.write().expect("write_pos lock poisoned");
        if *write_pos + entry_size > self.capacity {
            return Err(StoreError::Mmap {
                message: format!(
                    "mmap store full: need {} bytes, only {} available",
                    entry_size,
                    self.capacity - *write_pos
                ),
            });
        }

        let mut mmap = self.mmap.write().expect("mmap lock poisoned");
        let pos = *write_pos;
        mmap[pos..pos + 8].copy_from_slice(&key.get().to_le_bytes());
        mmap[pos + 8..pos + 12].copy_from_slice(&(value.len() as u32).to_le_bytes());
        mmap[pos + 12..pos + 12 + value.len()].copy_from_slice(value);

        let mut index = self.index.write().expect("index lock poisoned");
        index.insert(key, (pos + 12, value.len()));
        *write_pos = pos + entry_size;

        Ok(())
    }

    /// Read a value from the mmap store.
    pub fn get(&self, key: SymbolId) -> Option<Vec<u8>> {
        let index = self.index.read().expect("index lock poisoned");
        let &(offset, len) = index.get(&key)?;
        let mmap = self.mmap.read().expect("mmap lock poisoned");
        Some(mmap[offset..offset + len].to_vec())
    }

    /// Check if a key exists.
    pub fn contains(&self, key: SymbolId) -> bool {
        self.index
            .read()
            .expect("index lock poisoned")
            .contains_key(&key)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.index.read().expect("index lock poisoned").len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flush changes to disk.
    pub fn sync(&self) -> StoreResult<()> {
        self.mmap
            .read()
            .expect("mmap lock poisoned")
            .flush()
            .map_err(|e| StoreError::Io { source: e })
    }

    /// Path to the backing file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl std::fmt::Debug for MmapStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapStore")
            .field("path", &self.path)
            .field("capacity", &self.capacity)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn create_and_read_back() {
        let dir = TempDir::new().unwrap();
        let store = MmapStore::open_or_create(dir.path(), "test").unwrap();

        let id = SymbolId::new(1).unwrap();
        store.put(id, &[10, 20, 30]).unwrap();
        assert_eq!(store.get(id), Some(vec![10, 20, 30]));
        assert!(store.contains(id));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn reopen_preserves_data() {
        let dir = TempDir::new().unwrap();

        let id = SymbolId::new(42).unwrap();
        {
            let store = MmapStore::open_or_create(dir.path(), "persist").unwrap();
            store.put(id, &[1, 2, 3, 4]).unwrap();
            store.sync().unwrap();
        }

        // Reopen
        let store = MmapStore::open_or_create(dir.path(), "persist").unwrap();
        assert_eq!(store.get(id), Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn multiple_entries() {
        let dir = TempDir::new().unwrap();
        let store = MmapStore::open_or_create(dir.path(), "multi").unwrap();

        for i in 1..=100u64 {
            let id = SymbolId::new(i).unwrap();
            store.put(id, &[i as u8]).unwrap();
        }
        assert_eq!(store.len(), 100);

        for i in 1..=100u64 {
            let id = SymbolId::new(i).unwrap();
            assert_eq!(store.get(id), Some(vec![i as u8]));
        }
    }
}
