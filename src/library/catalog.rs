//! Persistent document catalog backed by `catalog.json`.
//!
//! The catalog is a simple JSON file listing all ingested documents.
//! It lives at `~/.local/share/akh-medu/library/catalog.json`.

use std::path::{Path, PathBuf};

use crate::library::error::{LibraryError, LibraryResult};
use crate::library::model::DocumentRecord;

/// Persistent index of all documents in the library.
pub struct LibraryCatalog {
    path: PathBuf,
    records: Vec<DocumentRecord>,
}

impl LibraryCatalog {
    /// Open or create a catalog at the given directory.
    ///
    /// The catalog file is `{dir}/catalog.json`. If it doesn't exist,
    /// starts with an empty list.
    pub fn open(dir: &Path) -> LibraryResult<Self> {
        let path = dir.join("catalog.json");

        let records = if path.exists() {
            let data = std::fs::read_to_string(&path).map_err(|e| LibraryError::CatalogIo {
                message: format!("read {}: {e}", path.display()),
            })?;
            serde_json::from_str(&data).map_err(|e| LibraryError::CatalogIo {
                message: format!("parse {}: {e}", path.display()),
            })?
        } else {
            Vec::new()
        };

        Ok(Self { path, records })
    }

    /// Flush the catalog to disk.
    fn flush(&self) -> LibraryResult<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| LibraryError::CatalogIo {
                message: format!("create dir {}: {e}", parent.display()),
            })?;
        }
        let json =
            serde_json::to_string_pretty(&self.records).map_err(|e| LibraryError::CatalogIo {
                message: format!("serialize catalog: {e}"),
            })?;
        std::fs::write(&self.path, json).map_err(|e| LibraryError::CatalogIo {
            message: format!("write {}: {e}", self.path.display()),
        })?;
        Ok(())
    }

    /// Add a document record. Returns error if the ID already exists.
    pub fn add(&mut self, record: DocumentRecord) -> LibraryResult<()> {
        if self.records.iter().any(|r| r.id == record.id) {
            return Err(LibraryError::Duplicate {
                id: record.id.clone(),
            });
        }
        self.records.push(record);
        self.flush()
    }

    /// Remove a document by ID. Returns the removed record, or error if not found.
    pub fn remove(&mut self, id: &str) -> LibraryResult<DocumentRecord> {
        let pos = self
            .records
            .iter()
            .position(|r| r.id == id)
            .ok_or_else(|| LibraryError::DocumentNotFound { id: id.into() })?;
        let record = self.records.remove(pos);
        self.flush()?;
        Ok(record)
    }

    /// Look up a document by ID.
    pub fn get(&self, id: &str) -> Option<&DocumentRecord> {
        self.records.iter().find(|r| r.id == id)
    }

    /// List all document records.
    pub fn list(&self) -> &[DocumentRecord] {
        &self.records
    }

    /// Number of documents in the catalog.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Generate a URL-safe slug from a title string.
pub fn slugify(title: &str) -> String {
    title
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_basic() {
        assert_eq!(slugify("Hello World"), "hello-world");
        assert_eq!(
            slugify("Rust Programming Language"),
            "rust-programming-language"
        );
        assert_eq!(slugify("  Multiple   Spaces  "), "multiple-spaces");
        assert_eq!(slugify("special!@#chars"), "special-chars");
    }

    #[test]
    fn catalog_add_and_list() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut catalog = LibraryCatalog::open(dir.path()).unwrap();
        assert!(catalog.is_empty());

        let record = DocumentRecord {
            id: "test-doc".into(),
            title: "Test Document".into(),
            source: crate::library::model::DocumentSource::Inline,
            format: crate::library::model::ContentFormat::Html,
            tags: vec!["test".into()],
            chunk_count: 5,
            triple_count: 10,
            ingested_at: 0,
        };
        catalog.add(record).unwrap();
        assert_eq!(catalog.len(), 1);
        assert!(catalog.get("test-doc").is_some());
    }

    #[test]
    fn catalog_duplicate_rejected() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut catalog = LibraryCatalog::open(dir.path()).unwrap();

        let record = DocumentRecord {
            id: "dup".into(),
            title: "Dup".into(),
            source: crate::library::model::DocumentSource::Inline,
            format: crate::library::model::ContentFormat::PlainText,
            tags: vec![],
            chunk_count: 0,
            triple_count: 0,
            ingested_at: 0,
        };
        catalog.add(record.clone()).unwrap();
        let err = catalog.add(record).unwrap_err();
        assert!(matches!(err, LibraryError::Duplicate { .. }));
    }

    #[test]
    fn catalog_remove() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut catalog = LibraryCatalog::open(dir.path()).unwrap();

        let record = DocumentRecord {
            id: "removable".into(),
            title: "Removable".into(),
            source: crate::library::model::DocumentSource::Inline,
            format: crate::library::model::ContentFormat::PlainText,
            tags: vec![],
            chunk_count: 0,
            triple_count: 0,
            ingested_at: 0,
        };
        catalog.add(record).unwrap();
        assert_eq!(catalog.len(), 1);

        let removed = catalog.remove("removable").unwrap();
        assert_eq!(removed.id, "removable");
        assert!(catalog.is_empty());
    }

    #[test]
    fn catalog_persists_across_reopen() {
        let dir = tempfile::TempDir::new().unwrap();

        {
            let mut catalog = LibraryCatalog::open(dir.path()).unwrap();
            let record = DocumentRecord {
                id: "persistent".into(),
                title: "Persistent".into(),
                source: crate::library::model::DocumentSource::File("/tmp/test.html".into()),
                format: crate::library::model::ContentFormat::Html,
                tags: vec!["test".into()],
                chunk_count: 3,
                triple_count: 7,
                ingested_at: 1234567890,
            };
            catalog.add(record).unwrap();
        }

        let catalog = LibraryCatalog::open(dir.path()).unwrap();
        assert_eq!(catalog.len(), 1);
        let rec = catalog.get("persistent").unwrap();
        assert_eq!(rec.title, "Persistent");
        assert_eq!(rec.chunk_count, 3);
    }
}
