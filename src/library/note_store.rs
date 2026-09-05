//! Note catalogs — the pre-Heka archival-search substrate (ADR 081 slice 7,
//! `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md` work-item 7).
//!
//! [`InMemoryNoteCatalog`] remains the lightweight test/bootstrap carrier.
//! [`PersistentNoteCatalog`] is the runtime path: per-akh, append-only,
//! content-addressed revisions with an atomic current-head view.
//!
//! Phase 42 (Heka/pgvector RAG) is **absent** on this branch (see
//! [`crate::agent::context::policy`]'s `RetrievalPolicy` deferral), so there is
//! no embedding index to search notes by vector similarity. Both catalogs use
//! the documented pre-Heka fallback: case-insensitive query-term overlap over
//! title + frontmatter + body, with an exact-phrase bonus. Swap this out for a
//! Heka-backed implementation of [`NoteCatalog`] once Phase 42 lands; the
//! archival tool depends only on the trait.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::library::error::{LibraryError, LibraryResult};
use crate::library::note::{Note, NoteId};

/// Read-only view over a collection of notes an akh can search.
///
/// **Invariant this trait exists to uphold**: a caller assembling *resident*
/// context (the core/recall tiers — see `agent::context::types::EpisodeState`)
/// must never be able to accidentally pull a note body into the working
/// context. [`Self::frontmatter_only`] physically cannot leak a body (bodies
/// are blanked before the `Note`s are returned); [`Self::search`] /
/// [`Self::get`] are the two explicit-retrieval paths where a body is allowed
/// to appear, and only because the caller asked for it by id or query.
pub trait NoteCatalog: Send + Sync {
    /// Every note with its body blanked — safe to fold into a resident
    /// (core/recall) context section without any risk of leaking a body.
    fn frontmatter_only(&self) -> Vec<Note>;

    /// Full note (frontmatter + body) by id — explicit retrieval by identity.
    fn get(&self, id: &NoteId) -> Option<Note>;

    /// Query-term search over title + frontmatter + body, returning full notes
    /// (bodies included) — the archival tier's on-demand path. The pre-Heka
    /// fallback ranks token overlap plus exact-phrase matches.
    fn search(&self, query: &str, top_k: usize) -> Vec<Note>;

    /// Metadata-bearing body-free catalog view. Implementations without
    /// durable version tokens may use the default `None` metadata; the
    /// persistent runtime catalog overrides this with its current head hash
    /// and timestamp.
    fn catalog_entries(&self) -> Vec<NoteCatalogEntry> {
        self.frontmatter_only()
            .into_iter()
            .map(|note| NoteCatalogEntry {
                note,
                revision: None,
                stored_at: None,
            })
            .collect()
    }

    /// Full note plus its current optimistic-concurrency token.
    fn get_entry(&self, id: &NoteId) -> Option<NoteCatalogEntry> {
        self.get(id).map(|note| NoteCatalogEntry {
            note,
            revision: None,
            stored_at: None,
        })
    }
}

/// A current note head together with the metadata needed to safely revise it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoteCatalogEntry {
    pub note: Note,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stored_at: Option<u64>,
}

/// Receipt for an append-only durable note revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteWriteReceipt {
    pub note_id: String,
    pub revision: String,
    pub stored_at: u64,
    /// `false` means the exact content revision already existed; no duplicate
    /// history record was written.
    pub created: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredNoteRevision {
    format_version: u32,
    revision: String,
    stored_at: u64,
    note: Note,
}

/// Durable, per-akh note catalog with append-only revision history.
///
/// `heads/<note-id>.json` is the fast current view. Every distinct body or
/// frontmatter state is first written to
/// `revisions/<note-id>/<content-hash>.json`; replacing a head therefore never
/// destroys the prior source material. Both files are written through a
/// same-directory temporary file and atomic rename. Re-appending byte-identical
/// content is idempotent.
pub struct PersistentNoteCatalog {
    root: PathBuf,
    heads: RwLock<BTreeMap<NoteId, StoredNoteRevision>>,
}

impl PersistentNoteCatalog {
    pub const FORMAT_VERSION: u32 = 1;

    /// Open (or create) a durable note catalog rooted at `root`.
    pub fn open(root: impl AsRef<Path>) -> LibraryResult<Self> {
        let root = root.as_ref().to_path_buf();
        let heads_dir = root.join("heads");
        let revisions_dir = root.join("revisions");
        fs::create_dir_all(&heads_dir).map_err(note_io)?;
        fs::create_dir_all(&revisions_dir).map_err(note_io)?;

        let mut heads = BTreeMap::new();
        let entries = fs::read_dir(&heads_dir).map_err(note_io)?;
        for entry in entries {
            let entry = entry.map_err(note_io)?;
            if !entry.file_type().map_err(note_io)?.is_file()
                || entry.path().extension().and_then(|x| x.to_str()) != Some("json")
            {
                continue;
            }
            let data = fs::read(entry.path()).map_err(note_io)?;
            let stored: StoredNoteRevision =
                serde_json::from_slice(&data).map_err(|e| LibraryError::CatalogIo {
                    message: format!("parse durable note head {}: {e}", entry.path().display()),
                })?;
            if stored.format_version != Self::FORMAT_VERSION {
                return Err(LibraryError::CatalogIo {
                    message: format!(
                        "unsupported note-store format {} in {} (expected {})",
                        stored.format_version,
                        entry.path().display(),
                        Self::FORMAT_VERSION
                    ),
                });
            }
            NoteId::parse(stored.note.id.to_string())?;
            let expected = revision_hash(&stored.note)?;
            if expected != stored.revision {
                return Err(LibraryError::CatalogIo {
                    message: format!(
                        "note head {} failed content-hash verification",
                        entry.path().display()
                    ),
                });
            }
            heads.insert(stored.note.id.clone(), stored);
        }

        Ok(Self {
            root,
            heads: RwLock::new(heads),
        })
    }

    /// Append a new immutable revision and advance the note's current head.
    /// The exact same content is a no-op with `created = false`.
    pub fn append_revision(&self, note: Note) -> LibraryResult<NoteWriteReceipt> {
        self.append_revision_checked(note, None)
    }

    /// Append a revision only if the caller observed the current head.
    ///
    /// `expected_revision` is an optimistic-concurrency token returned by
    /// [`NoteCatalog::get_entry`] / [`NoteCatalog::catalog_entries`]. Exact
    /// retries remain idempotent even when another process already completed
    /// the same write, while a stale write with different content fails
    /// without advancing the head.
    pub fn append_revision_checked(
        &self,
        note: Note,
        expected_revision: Option<&str>,
    ) -> LibraryResult<NoteWriteReceipt> {
        NoteId::parse(note.id.to_string())?;
        let revision = revision_hash(&note)?;

        let mut heads = self.heads.write().map_err(|_| LibraryError::CatalogIo {
            message: "durable note catalog lock poisoned".into(),
        })?;
        if let Some(current) = heads.get(&note.id)
            && current.revision == revision
        {
            return Ok(NoteWriteReceipt {
                note_id: note.id.to_string(),
                revision,
                stored_at: current.stored_at,
                created: false,
            });
        }
        if let Some(expected) = expected_revision {
            let actual = heads
                .get(&note.id)
                .map(|current| current.revision.as_str())
                .unwrap_or("<missing>");
            if actual != expected {
                return Err(LibraryError::NoteRevisionConflict {
                    note_id: note.id.to_string(),
                    expected: expected.to_string(),
                    actual: actual.to_string(),
                });
            }
        }

        let stored_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let stored = StoredNoteRevision {
            format_version: Self::FORMAT_VERSION,
            revision: revision.clone(),
            stored_at,
            note,
        };
        let bytes = serde_json::to_vec_pretty(&stored).map_err(|e| LibraryError::CatalogIo {
            message: format!("serialize durable note: {e}"),
        })?;

        let note_id = stored.note.id.to_string();
        let revision_dir = self.root.join("revisions").join(&note_id);
        fs::create_dir_all(&revision_dir).map_err(note_io)?;
        let revision_path = revision_dir.join(format!("{revision}.json"));
        if !revision_path.exists() {
            atomic_write(&revision_path, &bytes)?;
        }
        let head_path = self.root.join("heads").join(format!("{note_id}.json"));
        atomic_write(&head_path, &bytes)?;
        heads.insert(stored.note.id.clone(), stored);

        Ok(NoteWriteReceipt {
            note_id,
            revision,
            stored_at,
            created: true,
        })
    }

    /// Number of immutable revisions currently stored for `id`.
    pub fn revision_count(&self, id: &NoteId) -> LibraryResult<usize> {
        let dir = self.root.join("revisions").join(id.as_str());
        match fs::read_dir(&dir) {
            Ok(entries) => Ok(entries
                .filter_map(Result::ok)
                .filter(|e| {
                    e.file_type().map(|t| t.is_file()).unwrap_or(false)
                        && e.path().extension().and_then(|x| x.to_str()) == Some("json")
                })
                .count()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(e) => Err(note_io(e)),
        }
    }
}

impl NoteCatalog for PersistentNoteCatalog {
    fn frontmatter_only(&self) -> Vec<Note> {
        let Ok(heads) = self.heads.read() else {
            return Vec::new();
        };
        let mut values: Vec<&StoredNoteRevision> = heads.values().collect();
        values.sort_by(|a, b| {
            b.stored_at
                .cmp(&a.stored_at)
                .then_with(|| a.note.id.cmp(&b.note.id))
        });
        values
            .into_iter()
            .map(|stored| {
                let mut note = stored.note.clone();
                note.set_body(String::new());
                note
            })
            .collect()
    }

    fn get(&self, id: &NoteId) -> Option<Note> {
        self.heads
            .read()
            .ok()
            .and_then(|heads| heads.get(id).map(|stored| stored.note.clone()))
    }

    fn search(&self, query: &str, top_k: usize) -> Vec<Note> {
        if query.trim().is_empty() || top_k == 0 {
            return Vec::new();
        }
        let Ok(heads) = self.heads.read() else {
            return Vec::new();
        };
        let mut scored: Vec<(usize, u64, &Note)> = heads
            .values()
            .filter_map(|stored| {
                note_search_score(query, &stored.note)
                    .map(|score| (score, stored.stored_at, &stored.note))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
        scored
            .into_iter()
            .take(top_k)
            .map(|(_, _, note)| note.clone())
            .collect()
    }

    fn catalog_entries(&self) -> Vec<NoteCatalogEntry> {
        let Ok(heads) = self.heads.read() else {
            return Vec::new();
        };
        let mut values: Vec<&StoredNoteRevision> = heads.values().collect();
        values.sort_by(|a, b| {
            b.stored_at
                .cmp(&a.stored_at)
                .then_with(|| a.note.id.cmp(&b.note.id))
        });
        values
            .into_iter()
            .map(|stored| {
                let mut note = stored.note.clone();
                note.set_body(String::new());
                NoteCatalogEntry {
                    note,
                    revision: Some(stored.revision.clone()),
                    stored_at: Some(stored.stored_at),
                }
            })
            .collect()
    }

    fn get_entry(&self, id: &NoteId) -> Option<NoteCatalogEntry> {
        self.heads.read().ok().and_then(|heads| {
            heads.get(id).map(|stored| NoteCatalogEntry {
                note: stored.note.clone(),
                revision: Some(stored.revision.clone()),
                stored_at: Some(stored.stored_at),
            })
        })
    }
}

fn revision_hash(note: &Note) -> LibraryResult<String> {
    let bytes = serde_json::to_vec(note).map_err(|e| LibraryError::CatalogIo {
        message: format!("serialize note for content hash: {e}"),
    })?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn note_io(source: std::io::Error) -> LibraryError {
    LibraryError::CatalogIo {
        message: format!("durable note store: {source}"),
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn atomic_write(path: &Path, bytes: &[u8]) -> LibraryResult<()> {
    let parent = path.parent().ok_or_else(|| LibraryError::CatalogIo {
        message: format!("note-store path has no parent: {}", path.display()),
    })?;
    fs::create_dir_all(parent).map_err(note_io)?;
    let file_name =
        path.file_name()
            .and_then(|x| x.to_str())
            .ok_or_else(|| LibraryError::CatalogIo {
                message: format!("note-store path has no UTF-8 file name: {}", path.display()),
            })?;
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp = parent.join(format!(".{file_name}.tmp-{}-{nonce}", std::process::id()));
    let write_result = (|| -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    write_result.map_err(note_io)
}

/// Simple in-memory, thread-safe note catalog — the minimal carrier described
/// in the module docs.
#[derive(Default)]
pub struct InMemoryNoteCatalog {
    notes: RwLock<Vec<Note>>,
}

impl InMemoryNoteCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a catalog pre-populated with `notes` (test/bootstrap convenience).
    pub fn with_notes(notes: Vec<Note>) -> Self {
        Self {
            notes: RwLock::new(notes),
        }
    }

    fn lock(&self) -> std::sync::RwLockReadGuard<'_, Vec<Note>> {
        self.notes.read().expect("note catalog lock poisoned")
    }

    /// Insert a note, replacing any existing note with the same id.
    pub fn insert(&self, note: Note) {
        let mut notes = self.notes.write().expect("note catalog lock poisoned");
        if let Some(existing) = notes.iter_mut().find(|n| n.id == note.id) {
            *existing = note;
        } else {
            notes.push(note);
        }
    }

    /// Number of notes in the catalog.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the catalog holds no notes.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }
}

impl NoteCatalog for InMemoryNoteCatalog {
    fn frontmatter_only(&self) -> Vec<Note> {
        self.lock()
            .iter()
            .map(|n| {
                let mut blanked = n.clone();
                blanked.set_body(String::new());
                blanked
            })
            .collect()
    }

    fn get(&self, id: &NoteId) -> Option<Note> {
        self.lock().iter().find(|n| &n.id == id).cloned()
    }

    fn search(&self, query: &str, top_k: usize) -> Vec<Note> {
        if query.trim().is_empty() || top_k == 0 {
            return Vec::new();
        }
        let notes = self.lock();
        let mut scored: Vec<(usize, usize, &Note)> = notes
            .iter()
            .enumerate()
            .filter_map(|(order, n)| note_search_score(query, n).map(|score| (score, order, n)))
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored
            .into_iter()
            .take(top_k)
            .map(|(_, _, n)| n.clone())
            .collect()
    }
}

fn note_search_score(query: &str, note: &Note) -> Option<usize> {
    let query = query.trim().to_lowercase();
    if query.is_empty() {
        return None;
    }
    let haystack = format!(
        "{} {} {}",
        note.document.title,
        note.frontmatter.to_block(),
        note.body
    )
    .to_lowercase();
    let exact = haystack.matches(&query).count();
    let terms: std::collections::BTreeSet<String> = query
        .split(|character: char| {
            !character.is_alphanumeric() && character != '-' && character != '_'
        })
        .map(str::trim)
        .filter(|term| term.chars().count() >= 3)
        .map(String::from)
        .collect();
    let overlap: usize = terms
        .iter()
        .map(|term| haystack.matches(term).count().min(4))
        .sum();
    let score = exact.saturating_mul(100).saturating_add(overlap);
    (score > 0).then_some(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::model::{ContentFormat, DocumentRecord, DocumentSource};
    use crate::library::note::{Frontmatter, FrontmatterValue};

    fn doc(id: &str, title: &str) -> DocumentRecord {
        DocumentRecord {
            id: id.into(),
            title: title.into(),
            source: DocumentSource::Inline,
            format: ContentFormat::PlainText,
            tags: vec![],
            chunk_count: 0,
            triple_count: 0,
            ingested_at: 0,
        }
    }

    #[test]
    fn frontmatter_only_never_carries_a_body() {
        let catalog = InMemoryNoteCatalog::new();
        let mut fm = Frontmatter::new();
        fm.insert("title", FrontmatterValue::Text("Paris".into()));
        let note =
            Note::new(doc("doc-1", "Paris"), "SECRET BODY TEXT about Paris.").with_frontmatter(fm);
        catalog.insert(note);

        let blanked = catalog.frontmatter_only();
        assert_eq!(blanked.len(), 1);
        assert!(blanked[0].body.is_empty(), "body must be blanked");
        assert!(!blanked[0].frontmatter.is_empty(), "frontmatter survives");
    }

    #[test]
    fn get_returns_the_full_note_including_body() {
        let catalog = InMemoryNoteCatalog::new();
        let note = Note::new(doc("doc-1", "Paris"), "Paris is the capital of France.");
        let id = note.id.clone();
        catalog.insert(note);

        let full = catalog.get(&id).expect("note present");
        assert_eq!(full.body, "Paris is the capital of France.");
    }

    #[test]
    fn search_finds_matches_in_body_and_ranks_by_hit_count() {
        let catalog = InMemoryNoteCatalog::new();
        catalog.insert(Note::new(doc("a", "Rome"), "Rome is a city in Italy."));
        catalog.insert(Note::new(
            doc("b", "Paris"),
            "Paris is the capital of France. Paris has the Eiffel Tower.",
        ));
        catalog.insert(Note::new(
            doc("c", "Berlin"),
            "Berlin is a city in Germany.",
        ));

        let hits = catalog.search("paris", 5);
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].body.contains("Eiffel Tower"),
            "full body returned on explicit search"
        );
    }

    #[test]
    fn natural_language_question_retrieves_note_without_verbatim_match() {
        let catalog = InMemoryNoteCatalog::new();
        catalog.insert(Note::new(
            doc("keph", "Namar-delta-42 keph index"),
            "In edition 2, Namar-delta-42 has keph index QZ-731.",
        ));
        catalog.insert(Note::new(
            doc("other", "Other keph index"),
            "In edition 2, Velun-tor-18 has keph index AA-100.",
        ));

        let hits = catalog.search("What is the keph index of Namar-delta-42?", 1);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].body.contains("QZ-731"));
    }

    #[test]
    fn search_with_no_matches_is_empty() {
        let catalog = InMemoryNoteCatalog::new();
        catalog.insert(Note::new(doc("a", "Rome"), "Rome is a city in Italy."));
        assert!(catalog.search("nonexistent", 5).is_empty());
    }

    #[test]
    fn insert_replaces_existing_note_with_same_id() {
        let catalog = InMemoryNoteCatalog::new();
        let mut note = Note::new(doc("a", "Rome"), "First body.");
        let id = note.id.clone();
        catalog.insert(note.clone());
        note.set_body("Updated body.");
        catalog.insert(note);

        assert_eq!(catalog.len(), 1, "same id replaces, not duplicates");
        assert_eq!(catalog.get(&id).unwrap().body, "Updated body.");
    }

    #[test]
    fn persistent_catalog_survives_reopen_and_keeps_revision_history() {
        let tmp = tempfile::tempdir().unwrap();
        let note = Note::new(
            doc("textbook", "Systems Handbook"),
            "Version one: use a mutex.",
        );
        let id = note.id.clone();

        let catalog = PersistentNoteCatalog::open(tmp.path()).unwrap();
        let first = catalog.append_revision(note.clone()).unwrap();
        assert!(first.created);
        let duplicate = catalog.append_revision(note.clone()).unwrap();
        assert!(!duplicate.created);
        assert_eq!(first.revision, duplicate.revision);
        assert_eq!(catalog.revision_count(&id).unwrap(), 1);

        let mut revised = note;
        revised.set_body("Version two: use an RwLock for read-heavy access.");
        let second = catalog.append_revision(revised).unwrap();
        assert!(second.created);
        assert_ne!(first.revision, second.revision);
        assert_eq!(catalog.revision_count(&id).unwrap(), 2);
        drop(catalog);

        let reopened = PersistentNoteCatalog::open(tmp.path()).unwrap();
        let current = reopened.get(&id).expect("head survives restart");
        assert!(current.body.contains("RwLock"));
        assert_eq!(reopened.revision_count(&id).unwrap(), 2);
    }

    #[test]
    fn checked_append_rejects_a_stale_head_without_losing_history() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog = PersistentNoteCatalog::open(tmp.path()).unwrap();
        let note = Note::new(doc("checked", "Checked update"), "Version one.");
        let id = note.id.clone();
        let first = catalog.append_revision(note.clone()).unwrap();

        let mut second_note = note.clone();
        second_note.set_body("Version two.");
        let second = catalog
            .append_revision_checked(second_note.clone(), Some(&first.revision))
            .unwrap();

        let mut stale = note;
        stale.set_body("A stale competing version.");
        let error = catalog
            .append_revision_checked(stale, Some(&first.revision))
            .unwrap_err();
        assert!(matches!(error, LibraryError::NoteRevisionConflict { .. }));
        assert_eq!(catalog.get(&id).unwrap().body, "Version two.");
        assert_eq!(catalog.revision_count(&id).unwrap(), 2);

        let retry = catalog
            .append_revision_checked(second_note, Some(&first.revision))
            .unwrap();
        assert!(!retry.created, "an exact retry remains idempotent");
        assert_eq!(retry.revision, second.revision);
    }

    #[test]
    fn persistent_catalog_entries_expose_the_current_revision_token() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog = PersistentNoteCatalog::open(tmp.path()).unwrap();
        let note = Note::new(doc("versioned", "Versioned note"), "Full body.");
        let id = note.id.clone();
        let receipt = catalog.append_revision(note).unwrap();

        let listed = catalog.catalog_entries();
        assert_eq!(listed.len(), 1);
        assert!(listed[0].note.body.is_empty());
        assert_eq!(
            listed[0].revision.as_deref(),
            Some(receipt.revision.as_str())
        );
        assert_eq!(listed[0].stored_at, Some(receipt.stored_at));

        let read = catalog.get_entry(&id).unwrap();
        assert_eq!(read.note.body, "Full body.");
        assert_eq!(read.revision.as_deref(), Some(receipt.revision.as_str()));
    }

    #[test]
    fn persistent_frontmatter_view_never_leaks_body() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog = PersistentNoteCatalog::open(tmp.path()).unwrap();
        let mut fm = Frontmatter::new();
        fm.insert("type", FrontmatterValue::Text("TechArticle".into()));
        let note = Note::new(doc("secret", "Private Notes"), "body must stay archival")
            .with_frontmatter(fm);
        catalog.append_revision(note).unwrap();

        let resident = catalog.frontmatter_only();
        assert_eq!(resident.len(), 1);
        assert!(resident[0].body.is_empty());
        assert_eq!(
            resident[0].frontmatter.get("type"),
            Some(&FrontmatterValue::Text("TechArticle".into()))
        );
        let archival = catalog.search("archival", 3);
        assert_eq!(archival.len(), 1);
        assert!(archival[0].body.contains("body must stay archival"));
    }

    #[test]
    fn persistent_open_rejects_tampered_head() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog = PersistentNoteCatalog::open(tmp.path()).unwrap();
        let note = Note::new(doc("tamper", "Tamper Test"), "original");
        let id = note.id.clone();
        catalog.append_revision(note).unwrap();
        drop(catalog);

        let head = tmp.path().join("heads").join(format!("{id}.json"));
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&head).unwrap()).unwrap();
        value["note"]["body"] = serde_json::json!("tampered");
        std::fs::write(&head, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

        assert!(PersistentNoteCatalog::open(tmp.path()).is_err());
    }
}
