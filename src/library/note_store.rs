//! In-memory note catalog — the pre-Heka archival-search substrate (ADR 081
//! slice 7, `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md` work-item 7).
//!
//! [`Note`] ([`super::note`], ADR 081 work-item P1) shipped only the carrier
//! type — no persistent store. Slice 7 needs *something* the archival tier can
//! search over on demand, so this module adds the minimal carrier: a
//! thread-safe, in-memory catalog. A durable (disk/DB-backed) note store is a
//! follow-up once notes gain their own persistence story; this is deliberately
//! not that.
//!
//! Phase 42 (Heka/pgvector RAG) is **absent** on this branch (see
//! [`crate::agent::context::policy`]'s `RetrievalPolicy` deferral), so there is
//! no embedding index to search notes by vector similarity. [`InMemoryNoteCatalog::search`]
//! is the documented pre-Heka fallback: case-insensitive substring matching
//! over title + frontmatter + body, ranked by hit count. Swap this out for a
//! Heka-backed implementation of [`NoteCatalog`] once Phase 42 lands — nothing
//! else in the archival tier (`agent::tools::core_memory::ArchivalSearchTool`)
//! needs to change, since it depends only on the trait.

use std::sync::RwLock;

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

    /// Keyword/substring search over title + frontmatter + body, returning
    /// full notes (bodies included) — the archival tier's on-demand path.
    /// Pre-Heka fallback: case-insensitive substring match, ranked by hit
    /// count (most matches first), ties broken by insertion order.
    fn search(&self, query: &str, top_k: usize) -> Vec<Note>;
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
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return Vec::new();
        }
        let notes = self.lock();
        let mut scored: Vec<(usize, usize, &Note)> = notes
            .iter()
            .enumerate()
            .filter_map(|(order, n)| {
                let haystack = format!(
                    "{} {} {}",
                    n.document.title,
                    n.frontmatter.to_block(),
                    n.body
                )
                .to_lowercase();
                let hits = haystack.matches(&q).count();
                (hits > 0).then_some((hits, order, n))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.into_iter().take(top_k).map(|(_, _, n)| n.clone()).collect()
    }
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
        let note = Note::new(doc("doc-1", "Paris"), "SECRET BODY TEXT about Paris.")
            .with_frontmatter(fm);
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
        catalog.insert(Note::new(doc("c", "Berlin"), "Berlin is a city in Germany."));

        let hits = catalog.search("paris", 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].body.contains("Eiffel Tower"), "full body returned on explicit search");
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
}
